import React, { useRef, useMemo, useCallback, useState } from "react";
import { Tabs, useRouter } from "expo-router";
import { Platform, Pressable, TouchableOpacity } from "react-native";
import {
  Home,
  Plus,
  Settings,
  Grid3x3,
  User,
  MessageCircle,
  Edit3,
} from "@tamagui/lucide-icons";
import { Button, Text, View, XStack, YStack } from "tamagui";
import BottomSheet, {
  BottomSheetView,
  BottomSheetBackdrop,
} from "@gorhom/bottom-sheet";
import { BlurView } from "expo-blur";

export default function TabLayout() {
  const router = useRouter();
  const bottomSheetRef = useRef<BottomSheet>(null);
  const snapPoints = useMemo(() => ["30%"], []);

  const handleCreateTabPress = useCallback(() => {
    bottomSheetRef.current?.expand();
  }, []);

  const handleSheetChanges = useCallback((index: number) => {}, []);

  const handleCloseBottomSheet = useCallback(() => {
    bottomSheetRef.current?.close();
  }, []);

  const handleChatOption = useCallback(() => {
    console.log("🎯 Chat option pressed");
    console.log("🎯 Current router state:", router);

    // Close bottom sheet first
    bottomSheetRef.current?.close();

    // Navigate to chat
    console.log("🎯 Attempting navigation to /chat");
    router.push("/chat");
  }, [router]);

  const handleEditImageOption = useCallback(() => {
    console.log("🎯 Edit Image option pressed");
    console.log("🎯 Current router state:", router);

    // Close bottom sheet first
    bottomSheetRef.current?.close();

    // Navigate to edit-image
    console.log("🎯 Attempting navigation to /edit-image");
    router.push("/edit-image");
  }, [router]);

  // Custom backdrop component with blur
  const renderBackdrop = useCallback(
    (props: any) => (
      <BottomSheetBackdrop
        {...props}
        disappearsOnIndex={-1}
        appearsOnIndex={0}
        style={[props.style]}
      >
        <BlurView
          intensity={100}
          tint="dark"
          style={{
            width: "100%",
            height: "100%",
          }}
        />
      </BottomSheetBackdrop>
    ),
    []
  );

  return (
    <>
      <Tabs
        screenOptions={{
          headerShown: false,
          tabBarStyle: {
            backgroundColor: "#111111",
            borderTopColor: "#282828",
            borderTopWidth: 1,
            height: Platform.OS === "ios" ? 88 : 64,
            paddingBottom: Platform.OS === "ios" ? 32 : 8,
            paddingTop: 2,
          },
          tabBarActiveTintColor: "#ffffff",
          tabBarInactiveTintColor: "#666666",
          tabBarLabelStyle: {
            fontSize: 12,
            fontWeight: "500",
            marginTop: 4,
          },
        }}
      >
        <Tabs.Screen
          name="index"
          options={{
            title: "Feed",
            tabBarIcon: ({ color, size }) => (
              <Home size={size} color={color as any} />
            ),
          }}
        />
        <Tabs.Screen
          name="grid"
          options={{
            title: "Gallery",
            tabBarIcon: ({ color, size }) => (
              <Grid3x3 size={size} color={color as any} />
            ),
          }}
        />
        <Tabs.Screen
          name="create"
          options={{
            title: "Create",
            tabBarIcon: ({ color, size }) => (
              <Plus size={size} color={color as any} />
            ),
          }}
          listeners={{
            tabPress: (e) => {
              e.preventDefault();
              handleCreateTabPress();
            },
          }}
        />
        <Tabs.Screen
          name="profile"
          options={{
            title: "Profile",
            tabBarIcon: ({ color, size }) => (
              <User size={size} color={color as any} />
            ),
          }}
        />
        <Tabs.Screen
          name="settings"
          options={{
            title: "Settings",
            tabBarIcon: ({ color, size }) => (
              <Settings size={size} color={color as any} />
            ),
          }}
        />
      </Tabs>

      {/* Create Bottom Sheet */}
      <BottomSheet
        ref={bottomSheetRef}
        index={-1}
        snapPoints={snapPoints}
        onChange={handleSheetChanges}
        enablePanDownToClose={true}
        backgroundStyle={{ backgroundColor: "#181818" }}
        handleIndicatorStyle={{ backgroundColor: "#888888" }}
        backdropComponent={renderBackdrop}
      >
        <BottomSheetView
          style={{ flex: 1, paddingHorizontal: 20, paddingTop: 12 }}
        >
          <YStack
            gap="$4"
            alignItems="flex-start"
            justifyContent="center"
            flex={1}
          >
            {/* Header */}
            <YStack alignItems="flex-start" gap="$2">
              <Text color="white" fontSize="$5" fontWeight="600">
                Choose Creation Mode
              </Text>
            </YStack>

            {/* Options */}
            <YStack gap="$3" width="100%">
              {/* Chat Option */}
              <Pressable onPress={handleChatOption}>
                <XStack
                  gap="$3"
                  alignItems="center"
                  backgroundColor="#222222"
                  borderWidth={1}
                  borderColor="#444444"
                  borderRadius="$4"
                  paddingHorizontal="$3"
                  paddingVertical="$2"
                >
                  <MessageCircle size={24} color="#ffffff" />

                  <YStack flex={1} gap="$1">
                    <Text color="#ffffff" fontSize="$3" fontWeight="600">
                      Chat
                    </Text>
                    <Text color="#cccccc" fontSize="$3">
                      Generate images from text prompts with AI
                    </Text>
                  </YStack>
                </XStack>
              </Pressable>

              {/* Edit Image Option */}
              <TouchableOpacity onPress={handleEditImageOption}>
                <XStack
                  gap="$3"
                  alignItems="center"
                  backgroundColor="#222222"
                  borderWidth={1}
                  borderColor="#444444"
                  borderRadius="$4"
                  paddingHorizontal="$3"
                  paddingVertical="$2"
                >
                  <Edit3 size={24} color="#ffffff" />
                  <YStack flex={1} gap="$1">
                    <Text color="#ffffff" fontSize="$3" fontWeight="600">
                      Edit Image
                    </Text>
                    <Text color="#cccccc" fontSize="$3">
                      Modify existing images with AI-powered editing
                    </Text>
                  </YStack>
                </XStack>
              </TouchableOpacity>
            </YStack>
          </YStack>
        </BottomSheetView>
      </BottomSheet>
    </>
  );
}
